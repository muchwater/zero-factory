// src/places/dto/place-nearby.dto.ts
import { ApiProperty } from '@nestjs/swagger';
import { PlaceCategory, PlaceType, BrandType } from '@prisma/client';

export class PlaceNearbyDto {
  @ApiProperty({ description: '장소 ID', example: 1 })
  id: number;

  @ApiProperty({ description: '장소 이름', example: '제로웨이스트 카페' })
  name: string;

  @ApiProperty({ 
    description: '브랜드 (SUNHWA / UTURN)', 
    enum: BrandType,
    example: 'SUNHWA', 
    required: false 
  })
  brand?: BrandType | null;

  @ApiProperty({ description: '설명', example: '텀블러 대여/반납 가능', required: false })
  description?: string | null;

  @ApiProperty({ description: '주소', example: '대전광역시 유성구 대학로 291' })
  address: string;

  @ApiProperty({ description: '카테고리', enum: PlaceCategory, example: 'STORE' })
  category: PlaceCategory;

  @ApiProperty({
    description: '장소 타입 배열',
    enum: PlaceType,
    isArray: true,
    example: ['RENT', 'RETURN'],
  })
  types: PlaceType[];

  @ApiProperty({ description: '연락처', example: '010-1234-5678', required: false })
  contact?: string | null;

  @ApiProperty({ description: '거리 (m)', example: 123.4 })
  distance: number;

  @ApiProperty({ description: '위도', example: 36.3731 })
  lat?: number;

  @ApiProperty({ description: '경도', example: 127.362 })
  lng?: number;

  @ApiProperty({
    description: '위치 정보',
    example: { lat: 36.3731, lng: 127.362 },
    required: false,
  })
  location?: {
    lat: number;
    lng: number;
  };

  @ApiProperty({
    description: '오늘의 운영시간',
    example: { isClosed: false, openTime: '09:00', closeTime: '18:00', dayName: '월요일' },
  })
  todayHours: {
    isClosed: boolean;
    openTime?: string | null;
    closeTime?: string | null;
    dayName?: string | null;
  };
}
