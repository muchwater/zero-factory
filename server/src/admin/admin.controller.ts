import {
  Controller,
  Get,
  Put,
  Param,
  UseGuards,
  Logger,
  Body,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiHeader, ApiBody } from '@nestjs/swagger';
import { PlacesService } from '../places/places.service';
import { MembersService } from '../members/members.service';
import { AdminCodeGuard } from './admin-code.guard';
import { PlaceDto } from '../places/dto/place.dto';
import { ActivatePlaceDto } from './dto/activate-place.dto';

@ApiTags('admin')
@Controller('admin')
@UseGuards(AdminCodeGuard)
@ApiHeader({
  name: 'x-admin-code',
  description: 'Admin authentication code',
  required: true,
})
export class AdminController {
  private readonly logger = new Logger(AdminController.name);

  constructor(
    private readonly placesService: PlacesService,
    private readonly membersService: MembersService,
  ) {}

  @Get('places/pending')
  @ApiOperation({ summary: 'Get all pending places' })
  @ApiResponse({ status: 200, type: [PlaceDto] })
  async getPendingPlaces() {
    this.logger.log('📋 Getting all pending places');
    return this.placesService.getAllPlaces('PENDING');
  }

  @Put('places/:id/activate')
  @ApiOperation({ summary: 'Activate a pending place (승인 시 브랜드 지정 가능)' })
  @ApiResponse({ status: 200, type: PlaceDto })
  async activatePlace(
    @Param('id') id: string,
    @Body() body: ActivatePlaceDto,
  ) {
    const brand = body.brand;
    this.logger.log(`✅ Activating place ID: ${id}${brand ? ` with brand: ${brand}` : ''}`);
    return this.placesService.updatePlaceStatus(Number(id), 'ACTIVE', brand);
  }

  @Put('places/:id/reject')
  @ApiOperation({ summary: 'Reject a pending place (set to INACTIVE)' })
  @ApiResponse({ status: 200, type: PlaceDto })
  async rejectPlace(@Param('id') id: string) {
    this.logger.log(`❌ Rejecting place ID: ${id}`);
    return this.placesService.updatePlaceStatus(Number(id), 'INACTIVE');
  }

  // ===== 회원 관리 API =====

  @Get('members')
  @ApiOperation({ summary: '전체 회원 목록 조회' })
  @ApiResponse({ status: 200 })
  async getAllMembers() {
    this.logger.log('📋 Getting all members');
    return this.membersService.findAll();
  }

  @Put('members/:id/restrict')
  @ApiOperation({ summary: '회원 적립 제한 설정' })
  @ApiResponse({ status: 200 })
  async restrictMember(@Param('id') id: string) {
    this.logger.log(`🚫 Restricting member ID: ${id}`);
    return this.membersService.setReceiptRestriction(id, true);
  }

  @Put('members/:id/unrestrict')
  @ApiOperation({ summary: '회원 적립 제한 해제' })
  @ApiResponse({ status: 200 })
  async unrestrictMember(@Param('id') id: string) {
    this.logger.log(`✅ Unrestricting member ID: ${id}`);
    return this.membersService.setReceiptRestriction(id, false);
  }
}
