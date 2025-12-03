import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class MembersService {
  constructor(private prisma: PrismaService) {}

  async findOrCreate(nickname: string, deviceId?: string) {
    // nickname으로 찾기 (unique이므로 findUnique 사용 가능)
    let member = await this.prisma.member.findUnique({
      where: { nickname },
    });

    if (!member && deviceId) {
      // nickname으로 찾지 못했으면, deviceId로 한 번 더 확인
      member = await this.prisma.member.findUnique({
        where: { deviceId },
      });

      if (member) {
        // deviceId로 찾았다면 nickname 업데이트
        member = await this.prisma.member.update({
          where: { id: member.id },
          data: { nickname },
        });
      }
    }

    if (!member) {
      // 둘 다 없으면 새로 생성
      member = await this.prisma.member.create({
        data: {
          nickname,
          deviceId: deviceId,
        },
      });
    }

    return member;
  }

  async findById(memberId: string) {
    const member = await this.prisma.member.findUnique({
      where: { id: memberId },
    });

    if (!member) {
      throw new NotFoundException(`Member with ID ${memberId} not found`);
    }

    return member;
  }

  /**
   * 전체 회원 목록 조회 (관리자용)
   * 최근 3일 적립 수를 포함하여 이상 적립 행동 감지에 활용
   */
  async findAll() {
    const threeDaysAgo = new Date();
    threeDaysAgo.setDate(threeDaysAgo.getDate() - 3);

    const members = await this.prisma.member.findMany({
      orderBy: { createdAt: 'desc' },
      select: {
        id: true,
        nickname: true,
        pointBalance: true,
        lastReceiptAt: true,
        receiptRestricted: true,
        createdAt: true,
        _count: {
          select: { receipts: true },
        },
      },
    });

    // 최근 3일 적립 수를 별도 쿼리로 조회
    const receipts3DaysData = await this.prisma.receipt.groupBy({
      by: ['memberId'],
      where: {
        createdAt: {
          gte: threeDaysAgo,
        },
      },
      _count: {
        id: true,
      },
    });

    // memberId별 최근 3일 적립 수 맵 생성
    const receipts3DaysMap = new Map(
      receipts3DaysData.map((item) => [item.memberId, item._count.id]),
    );

    // 회원 데이터에 최근 3일 적립 수 추가
    return members.map((member) => ({
      ...member,
      receipts3Days: receipts3DaysMap.get(member.id) || 0,
    }));
  }

  /**
   * 회원 적립 제한 설정/해제
   */
  async setReceiptRestriction(memberId: string, restricted: boolean) {
    const member = await this.prisma.member.findUnique({
      where: { id: memberId },
    });

    if (!member) {
      throw new NotFoundException(`Member with ID ${memberId} not found`);
    }

    return this.prisma.member.update({
      where: { id: memberId },
      data: { receiptRestricted: restricted },
    });
  }
}
