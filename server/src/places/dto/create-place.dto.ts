import { ApiProperty } from '@nestjs/swagger';
import { PlaceCategory, PlaceType } from '@prisma/client';
import { IsString, IsEnum, IsArray, IsOptional, IsNumber, ValidateNested } from 'class-validator';
import { Type } from 'class-transformer';

export class LocationDto {
  @ApiProperty({ description: '위도', example: 36.3731 })
  @IsNumber()
  lat: number;

  @ApiProperty({ description: '경도', example: 127.362 })
  @IsNumber()
  lng: number;
}

export class CreatePlaceDto {
  @ApiProperty({ description: '장소 이름', example: '제로웨이스트 카페' })
  @IsString()
  name: string;

  @ApiProperty({ description: '장소 설명', example: '텀블러 대여/반납 가능', required: false })
  @IsOptional()
  @IsString()
  description?: string;

  @ApiProperty({ description: '주소', example: '대전광역시 유성구 대학로 291' })
  @IsString()
  address: string;

  @ApiProperty({ description: '상세 주소', example: '1층 카페', required: false })
  @IsOptional()
  @IsString()
  detailAddress?: string;

  @ApiProperty({
    description: '카테고리 (STORE / FACILITY)',
    enum: PlaceCategory,
    example: 'FACILITY',
  })
  @IsEnum(PlaceCategory)
  category: PlaceCategory;

  @ApiProperty({
    description: '장소 타입 배열 (RENT, RETURN, BONUS, CLEAN)',
    enum: PlaceType,
    isArray: true,
    example: ['RENT', 'RETURN'],
  })
  @IsArray()
  @IsEnum(PlaceType, { each: true })
  types: PlaceType[];

  @ApiProperty({ description: '연락처', example: '010-1234-5678', required: false })
  @IsOptional()
  @IsString()
  contact?: string;

  @ApiProperty({ description: '제보된 서비스명 (리유저블 컨테이너/RVM)', example: '선화', required: false })
  @IsOptional()
  @IsString()
  reportedBrand?: string;

  @ApiProperty({ type: LocationDto, description: '위치 정보 (위도/경도)' })
  @ValidateNested()
  @Type(() => LocationDto)
  location: LocationDto;
}
