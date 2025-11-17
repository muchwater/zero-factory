import { Injectable, BadRequestException, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { TransactionType } from '@prisma/client';
import dayjs from 'dayjs';
import { GetTransactionsQueryDto } from './dto/get-transactions-query.dto';
import { TransactionHistoryResponseDto } from './dto/transaction-history-response.dto';

@Injectable()
export class PointsService {
  constructor(private prisma: PrismaService) {}

  /**
   * 포인트 적립
   */
  async earnPoints(memberId: string, amount: number, placeId: number) {
    const member = await this.prisma.member.findUnique({
      where: { id: memberId },
    });

    if (!member) {
      throw new NotFoundException('회원을 찾을 수 없습니다.');
    }

    // PointTransaction 생성
    const transaction = await this.prisma.pointTransaction.create({
      data: {
        memberId,
        placeId,
        amount,
        type: TransactionType.EARN,
      },
    });

    // PointLog 업데이트
    await this.updatePointLog(memberId, amount, TransactionType.EARN);

    // Member의 pointBalance 업데이트
    await this.updateMemberBalance(memberId);

    return transaction;
  }

  /**
   * 포인트 사용
   */
  async redeemPoints(memberId: string, amount: number, placeId: number) {
    const member = await this.prisma.member.findUnique({
      where: { id: memberId },
    });

    if (!member) {
      throw new NotFoundException('회원을 찾을 수 없습니다.');
    }

    // 유효한 잔액 확인
    const { balance } = await this.getValidBalance(memberId);

    if (balance < amount) {
      throw new BadRequestException(
        `포인트가 부족합니다. 현재 잔액: ${balance}, 사용 요청: ${amount}`,
      );
    }

    // PointTransaction 생성
    const transaction = await this.prisma.pointTransaction.create({
      data: {
        memberId,
        placeId,
        amount,
        type: TransactionType.REDEEM,
      },
    });

    // PointLog 업데이트
    await this.updatePointLog(memberId, amount, TransactionType.REDEEM);

    // Member의 pointBalance 업데이트
    await this.updateMemberBalance(memberId);

    return transaction;
  }

  /**
   * 유효한 포인트 잔액 조회 (12개월 이내)
   */
  async getValidBalance(memberId: string) {
    const member = await this.prisma.member.findUnique({
      where: { id: memberId },
    });

    if (!member) {
      throw new NotFoundException('회원을 찾을 수 없습니다.');
    }

    const now = dayjs();
    const currentYearMonth = now.format('YYYYMM');
    const twelveMonthsAgo = now.subtract(11, 'month').format('YYYYMM');

    // 최근 12개월 PointLog 조회
    const pointLogs = await this.prisma.pointLog.findMany({
      where: {
        memberId,
        yearMonth: {
          gte: twelveMonthsAgo,
          lte: currentYearMonth,
        },
      },
    });

    // 12개월 이내 포인트 합산
    const balance = pointLogs.reduce((sum, log) => {
      return sum + log.earnedPoints - log.redeemedPoints;
    }, 0);

    // 다음달 1일에 소멸될 포인트 계산 (13개월 전 PointLog)
    const thirteenMonthsAgo = now.subtract(12, 'month').format('YYYYMM');
    const expiringLog = await this.prisma.pointLog.findFirst({
      where: {
        memberId,
        yearMonth: thirteenMonthsAgo,
      },
    });

    const expiringNextMonth = expiringLog
      ? Math.max(0, expiringLog.earnedPoints - expiringLog.redeemedPoints)
      : 0;

    return {
      balance,
      expiringNextMonth,
    };
  }

  /**
   * PointLog 업데이트
   */
  private async updatePointLog(
    memberId: string,
    amount: number,
    type: TransactionType,
  ) {
    const now = dayjs();
    const yearMonth = now.format('YYYYMM');

    // 해당 월의 PointLog 조회 또는 생성
    let pointLog = await this.prisma.pointLog.findFirst({
      where: {
        memberId,
        yearMonth,
      },
    });

    if (!pointLog) {
      // PointLog 생성
      const previousBalance = await this.calculatePreviousBalance(memberId, yearMonth);

      pointLog = await this.prisma.pointLog.create({
        data: {
          memberId,
          yearMonth,
          sumBalance: previousBalance + (type === TransactionType.EARN ? amount : -amount),
          earnedPoints: type === TransactionType.EARN ? amount : 0,
          redeemedPoints: type === TransactionType.REDEEM ? amount : 0,
        },
      });
    } else {
      // PointLog 업데이트
      const updateData =
        type === TransactionType.EARN
          ? {
              sumBalance: pointLog.sumBalance + amount,
              earnedPoints: pointLog.earnedPoints + amount,
            }
          : {
              sumBalance: pointLog.sumBalance - amount,
              redeemedPoints: pointLog.redeemedPoints + amount,
            };

      pointLog = await this.prisma.pointLog.update({
        where: { id: pointLog.id },
        data: updateData,
      });
    }

    return pointLog;
  }

  /**
   * 이전 월까지의 누적 잔액 계산 (12개월 이내)
   */
  private async calculatePreviousBalance(memberId: string, currentYearMonth: string) {
    const now = dayjs(currentYearMonth, 'YYYYMM');
    const twelveMonthsAgo = now.subtract(11, 'month').format('YYYYMM');
    const previousMonth = now.subtract(1, 'month').format('YYYYMM');

    const pointLogs = await this.prisma.pointLog.findMany({
      where: {
        memberId,
        yearMonth: {
          gte: twelveMonthsAgo,
          lte: previousMonth,
        },
      },
    });

    return pointLogs.reduce((sum, log) => {
      return sum + log.earnedPoints - log.redeemedPoints;
    }, 0);
  }

  /**
   * Member의 pointBalance 업데이트
   */
  private async updateMemberBalance(memberId: string) {
    const { balance } = await this.getValidBalance(memberId);

    await this.prisma.member.update({
      where: { id: memberId },
      data: { pointBalance: balance },
    });
  }

  /**
   * 포인트 거래 내역 조회 (페이지네이션)
   */
  async getTransactionHistory(
    memberId: string,
    query: GetTransactionsQueryDto,
  ): Promise<TransactionHistoryResponseDto> {
    const member = await this.prisma.member.findUnique({
      where: { id: memberId },
    });

    if (!member) {
      throw new NotFoundException('회원을 찾을 수 없습니다.');
    }

    const { page = 1, limit = 20, type } = query;
    const skip = (page - 1) * limit;

    // 필터 조건
    const where = {
      memberId,
      ...(type && { type }),
    };

    // 총 거래 건수 조회
    const totalCount = await this.prisma.pointTransaction.count({ where });

    // 거래 내역 조회 (최신순)
    const transactions = await this.prisma.pointTransaction.findMany({
      where,
      include: {
        place: {
          select: {
            id: true,
            name: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
      skip,
      take: limit,
    });

    // 응답 DTO 변환
    const transactionItems = transactions.map((transaction) => ({
      id: transaction.id,
      placeId: transaction.placeId,
      placeName: transaction.place.name,
      amount: transaction.amount,
      type: transaction.type,
      createdAt: transaction.createdAt,
    }));

    const totalPages = Math.ceil(totalCount / limit);

    return {
      transactions: transactionItems,
      totalCount,
      currentPage: page,
      pageSize: limit,
      totalPages,
    };
  }
}
