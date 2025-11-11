import { ApiProperty } from '@nestjs/swagger';
import { IsEnum, IsOptional } from 'class-validator';

export class UpdatePlaceStatusDto {
  @ApiProperty({
    description: '장소 상태',
    enum: ['ACTIVE', 'INACTIVE'],
    example: 'ACTIVE',
  })
  @IsEnum(['ACTIVE', 'INACTIVE'])
  status: 'ACTIVE' | 'INACTIVE';

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
