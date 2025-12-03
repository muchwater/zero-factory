import { IsString, IsNotEmpty, IsOptional, IsInt } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class CreateReceiptDto {
  @ApiProperty({ description: '제품/서비스 설명', example: 'One Americano' })
  @IsString()
  @IsNotEmpty()
  productDescription: string;

  @ApiProperty({ description: '장소 ID (선택)', required: false })
  @IsInt()
  @IsOptional()
  placeId?: number;

  @ApiProperty({ description: 'AI 검증 결과 JSON', required: false })
  @IsString()
  @IsOptional()
  verificationResult?: string;
}
