import { ApiProperty } from '@nestjs/swagger';
import { PlaceCategory, PlaceType, BrandType } from '@prisma/client';
import { StoreOpeningHourDto } from './store-opening-hour.dto';
import { StoreOpeningHourExceptionDto } from './store-opening-hour-exception.dto';

export class PlaceDto {
  @ApiProperty({ description: '장소 ID', example: 1 })
  id: number;

  @ApiProperty({ description: '장소 이름', example: '제로웨이스트 카페' })
  name: string;

  @ApiProperty({ description: '장소 설명', example: '텀블러 대여/반납 가능', required: false })
  description?: string | null;

  @ApiProperty({ description: '주소', example: '대전광역시 유성구 대학로 291' })
  address: string;

  @ApiProperty({
    description: '카테고리 (STORE / FACILITY)',
    enum: PlaceCategory,
    example: 'STORE',
  })
  category: PlaceCategory;

  @ApiProperty({
    description: '장소 타입 배열 (RENT, RETURN, BONUS, CLEAN)',
    enum: PlaceType,
    isArray: true,
    example: ['RENT', 'RETURN'],
  })
  types: PlaceType[];

  @ApiProperty({ description: '연락처 (Store 전용)', example: '010-1234-5678', required: false })
  contact?: string | null;

  @ApiProperty({ description: '제보자가 입력한 서비스명 (자유 텍스트)', example: '선화점', required: false })
  reportedBrand?: string | null;

  @ApiProperty({ 
    description: '관리자가 승인 시 선택한 브랜드 (SUNHWA / UTURN)', 
    enum: BrandType,
    example: 'SUNHWA',
    required: false 
  })
  brand?: BrandType | null;

  @ApiProperty({ description: '생성 시각', example: '2025-02-01T09:00:00.000Z' })
  createdAt: Date;

  @ApiProperty({ description: '업데이트 시각', example: '2025-02-01T12:30:00.000Z' })
  updatedAt: Date;

  @ApiProperty({ type: [StoreOpeningHourDto], description: '요일별 기본 운영시간' })
  openingHours: StoreOpeningHourDto[];

  @ApiProperty({ type: [StoreOpeningHourExceptionDto], description: '예외 운영시간 규칙' })
  exceptions: StoreOpeningHourExceptionDto[];
}
