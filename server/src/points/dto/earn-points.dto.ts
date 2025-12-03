import { ApiProperty } from '@nestjs/swagger';
import { IsInt, IsPositive, IsOptional } from 'class-validator';

export class EarnPointsDto {
  @ApiProperty({ description: '적립할 포인트', example: 100 })
  @IsInt()
  @IsPositive()
  amount: number;

  @ApiProperty({ description: '포인트 적립 장소 ID', example: 1, required: false })
  @IsInt()
  @IsPositive()
  @IsOptional()
  placeId?: number;
}
