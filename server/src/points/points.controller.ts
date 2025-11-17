import { Controller, Post, Get, Body, Param, Query } from '@nestjs/common';
import { PointsService } from './points.service';
import { EarnPointsDto } from './dto/earn-points.dto';
import { RedeemPointsDto } from './dto/redeem-points.dto';
import { PointBalanceResponseDto } from './dto/point-balance-response.dto';
import { GetTransactionsQueryDto } from './dto/get-transactions-query.dto';
import { TransactionHistoryResponseDto } from './dto/transaction-history-response.dto';
import { ApiTags, ApiOperation, ApiResponse, ApiParam } from '@nestjs/swagger';

@ApiTags('points')
@Controller('members/:memberId/points')
export class PointsController {
  constructor(private readonly pointsService: PointsService) {}

  @Post('earn')
  @ApiOperation({
    summary: '포인트 적립',
    description: '포인트를 적립합니다. PointTransaction과 PointLog가 업데이트됩니다.',
  })
  @ApiParam({ name: 'memberId', description: '회원 ID', example: 'uuid-string' })
  @ApiResponse({ status: 201, description: '포인트 적립 성공' })
  @ApiResponse({ status: 404, description: '회원을 찾을 수 없음' })
  async earnPoints(
    @Param('memberId') memberId: string,
    @Body() dto: EarnPointsDto,
  ) {
    return this.pointsService.earnPoints(memberId, dto.amount, dto.placeId);
  }

  @Post('redeem')
  @ApiOperation({
    summary: '포인트 사용',
    description: '포인트를 사용합니다. 잔액이 부족하면 에러를 반환합니다.',
  })
  @ApiParam({ name: 'memberId', description: '회원 ID', example: 'uuid-string' })
  @ApiResponse({ status: 201, description: '포인트 사용 성공' })
  @ApiResponse({ status: 400, description: '포인트 잔액 부족' })
  @ApiResponse({ status: 404, description: '회원을 찾을 수 없음' })
  async redeemPoints(
    @Param('memberId') memberId: string,
    @Body() dto: RedeemPointsDto,
  ) {
    return this.pointsService.redeemPoints(memberId, dto.amount, dto.placeId);
  }

  @Get('balance')
  @ApiOperation({
    summary: '유효한 포인트 잔액 조회',
    description:
      '12개월 이내의 포인트 잔액과 다음달 1일에 소멸될 포인트를 조회합니다.',
  })
  @ApiParam({ name: 'memberId', description: '회원 ID', example: 'uuid-string' })
  @ApiResponse({
    status: 200,
    description: '포인트 잔액 조회 성공',
    type: PointBalanceResponseDto,
  })
  @ApiResponse({ status: 404, description: '회원을 찾을 수 없음' })
  async getBalance(@Param('memberId') memberId: string): Promise<PointBalanceResponseDto> {
    return this.pointsService.getValidBalance(memberId);
  }

  @Get('transactions')
  @ApiOperation({
    summary: '포인트 거래 내역 조회',
    description:
      '회원의 포인트 적립/사용 내역을 조회합니다. 페이지네이션과 타입 필터를 지원합니다.',
  })
  @ApiParam({ name: 'memberId', description: '회원 ID', example: 'uuid-string' })
  @ApiResponse({
    status: 200,
    description: '거래 내역 조회 성공',
    type: TransactionHistoryResponseDto,
  })
  @ApiResponse({ status: 404, description: '회원을 찾을 수 없음' })
  async getTransactionHistory(
    @Param('memberId') memberId: string,
    @Query() query: GetTransactionsQueryDto,
  ): Promise<TransactionHistoryResponseDto> {
    return this.pointsService.getTransactionHistory(memberId, query);
  }
}
