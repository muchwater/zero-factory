import { IsString, IsOptional, IsInt } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';
import { Type } from 'class-transformer';

export class CreateReceiptDto {
  @ApiProperty({ description: '제품/서비스 설명 (선택)', example: 'One Americano', required: false })
  @IsString()
  @IsOptional()
  productDescription?: string;

  @ApiProperty({ description: '장소 ID (선택)', required: false })
  @Type(() => Number)
  @IsInt()
  @IsOptional()
  placeId?: number;

  @ApiProperty({ description: 'AI 검증 결과 JSON', required: false })
  @IsString()
  @IsOptional()
  verificationResult?: string;
}
