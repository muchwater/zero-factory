import { ApiProperty } from '@nestjs/swagger';
import { TransactionType } from '@prisma/client';

export class TransactionItemDto {
  @ApiProperty({ description: '거래 ID', example: 1 })
  id: number;

  @ApiProperty({ description: '거래 장소 ID', example: 5 })
  placeId: number;

  @ApiProperty({ description: '장소명', example: '스타벅스 강남점' })
  placeName: string;

  @ApiProperty({ description: '포인트 금액', example: 50 })
  amount: number;

  @ApiProperty({
    description: '거래 유형',
    enum: TransactionType,
    example: TransactionType.EARN,
  })
  type: TransactionType;

  @ApiProperty({ description: '거래 일시', example: '2025-11-17T10:30:00.000Z' })
  createdAt: Date;
}

export class TransactionHistoryResponseDto {
  @ApiProperty({ description: '거래 내역 목록', type: [TransactionItemDto] })
  transactions: TransactionItemDto[];

  @ApiProperty({ description: '총 거래 건수', example: 45 })
  totalCount: number;

  @ApiProperty({ description: '현재 페이지', example: 1 })
  currentPage: number;

  @ApiProperty({ description: '페이지당 항목 수', example: 20 })
  pageSize: number;

  @ApiProperty({ description: '총 페이지 수', example: 3 })
  totalPages: number;
}
