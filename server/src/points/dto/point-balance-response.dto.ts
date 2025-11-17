import { ApiProperty } from '@nestjs/swagger';

export class PointBalanceResponseDto {
  @ApiProperty({ description: '유효한 포인트 잔액 (12개월 이내)', example: 150 })
  balance: number;

  @ApiProperty({ description: '다음달 소멸 예정 포인트', example: 20 })
  expiringNextMonth: number;
}
