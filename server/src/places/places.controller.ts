import { Controller, Get, Param, Query } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiQuery } from '@nestjs/swagger';
import { PlacesService } from './places.service';
import { PlaceDto } from './dto/place.dto';
import { PlaceNearbyDto } from './dto/place-nearby.dto';

@ApiTags('places')
@Controller('places')
export class PlacesController {
  constructor(private readonly placesService: PlacesService) {}

  @Get()
  @ApiOperation({ summary: '장소 전체 조회' })
  @ApiResponse({ status: 200, type: [PlaceDto] })
  async getAllPlaces() {
    return this.placesService.getAllPlaces();
  }

  @Get('nearby')
  @ApiOperation({ summary: '좌표 기준 반경 내 장소 검색' })
  @ApiQuery({ name: 'lat', type: Number, example: 36.3731 })
  @ApiQuery({ name: 'lng', type: Number, example: 127.362 })
  @ApiQuery({ name: 'radius', type: Number, example: 100, description: '미터' })
  @ApiQuery({ name: 'limit', type: Number, required: false, example: 10 })
  @ApiQuery({ name: 'offset', type: Number, required: false, example: 0 })
  @ApiQuery({
    name: 'types',
    type: String,
    required: false,
    example: 'RENT,RETURN',
    description: '필터링할 PlaceType (콤마 구분) (RENT, RETURN, BONUS, CLEAN)',
  })
  @ApiResponse({ status: 200, type: [PlaceNearbyDto] })
  async getPlacesNearby(
    @Query('lat') lat: string,
    @Query('lng') lng: string,
    @Query('radius') radius: string,
    @Query('limit') limit = '10',
    @Query('offset') offset = '0',
    @Query('types') types?: string,
  ) {
    return this.placesService.getPlacesNearby(
      Number(lat),
      Number(lng),
      Number(radius),
      Number(limit),
      Number(offset),
      types ? types.split(',').map((t) => t.trim().toUpperCase()) : undefined,
    );
  }

  @Get(':id')
  @ApiOperation({ summary: '특정 장소 상세 조회' })
  @ApiResponse({ status: 200, type: PlaceDto })
  async getPlaceById(@Param('id') id: string) {
    return this.placesService.getPlaceById(Number(id));
  }
}
