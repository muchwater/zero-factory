import { ApiProperty } from '@nestjs/swagger';
import { IsEnum, IsOptional } from 'class-validator';
import { Type } from 'class-transformer';

export class ActivatePlaceDto {
  @ApiProperty({
    description: '브랜드 (선택사항)',
    enum: ['SUNHWA', 'UTURN'],
    example: 'SUNHWA',
    required: false,
  })
  @IsOptional()
  @IsEnum(['SUNHWA', 'UTURN'])
  brand?: 'SUNHWA' | 'UTURN';
}
