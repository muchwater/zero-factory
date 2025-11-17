import { ApiProperty } from '@nestjs/swagger';
import { IsInt, IsPositive } from 'class-validator';

export class RedeemPointsDto {
  @ApiProperty({ description: '사용할 포인트', example: 50 })
  @IsInt()
  @IsPositive()
  amount: number;

  @ApiProperty({ description: '포인트 사용 장소 ID', example: 1 })
  @IsInt()
  @IsPositive()
  placeId: number;
}
