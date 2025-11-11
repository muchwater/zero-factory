import { Controller, Get, Post, Param, Query, Body, Logger, Put, Headers } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiQuery } from '@nestjs/swagger';
import { PlacesService } from './places.service';
import { PlaceDto } from './dto/place.dto';
import { PlaceNearbyDto } from './dto/place-nearby.dto';
import { CreatePlaceDto } from './dto/create-place.dto';
import { UpdatePlaceStatusDto } from './dto/update-place-status.dto';

@ApiTags('places')
@Controller('places')
export class PlacesController {
  private readonly logger = new Logger(PlacesController.name);
  
  constructor(private readonly placesService: PlacesService) {}

  @Get()
  @ApiOperation({ summary: '장소 전체 조회' })
  @ApiQuery({ name: 'state', type: String, required: false, example: 'ACTIVE', description: '장소 상태 필터링 (ACTIVE, INACTIVE)' })
  @ApiResponse({ status: 200, type: [PlaceDto] })
  async getAllPlaces(@Query('state') state?: string) {
    return this.placesService.getAllPlaces(state);
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
    this.logger.log(`📍 GET /places/nearby - lat=${lat}, lng=${lng}, radius=${radius}`);
    
    const result = await this.placesService.getPlacesNearby(
      Number(lat),
      Number(lng),
      Number(radius),
      Number(limit),
      Number(offset),
      types ? types.split(',').map((t) => t.trim().toUpperCase()) : undefined,
    );
    
    this.logger.log(`✅ Found ${result.length} places`);
    return result;
  }

  @Post()
  @ApiOperation({ summary: '새 장소 제보' })
  @ApiResponse({ status: 201, type: PlaceDto })
  async createPlace(@Body() createPlaceDto: CreatePlaceDto) {
    this.logger.log(`📍 POST /places - Creating new place: ${createPlaceDto.name}`);
    return this.placesService.createPlace(createPlaceDto);
  }

  @Get(':id')
  @ApiOperation({ summary: '특정 장소 상세 조회' })
  @ApiResponse({ status: 200, type: PlaceDto })
  async getPlaceById(@Param('id') id: string) {
    return this.placesService.getPlaceById(Number(id));
  }

  @Put('status/:id')
  @ApiOperation({ summary: '특정 장소 상태 업데이트 (승인 시 브랜드 지정)' })
  @ApiResponse({ status: 200, type: PlaceDto })
  async updatePlaceStatus(
    @Param('id') id: string,
    @Body() body: UpdatePlaceStatusDto,
  ) {
    // body 객체에서 status와 brand 추출 (폴백)
    const status = (body as any)?.status as 'ACTIVE' | 'INACTIVE';
    const brand = (body as any)?.brand as 'SUNHWA' | 'UTURN' | undefined;
    this.logger.log(`🔄 PUT /places/status/${id} - status=${status}, brand=${brand}`);
    return this.placesService.updatePlaceStatus(Number(id), status, brand);
  }
}
