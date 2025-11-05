import {
  Controller,
  Get,
  Put,
  Param,
  UseGuards,
  Logger,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiHeader } from '@nestjs/swagger';
import { PlacesService } from '../places/places.service';
import { AdminCodeGuard } from './admin-code.guard';
import { PlaceDto } from '../places/dto/place.dto';

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

  constructor(private readonly placesService: PlacesService) {}

  @Get('places/pending')
  @ApiOperation({ summary: 'Get all pending places' })
  @ApiResponse({ status: 200, type: [PlaceDto] })
  async getPendingPlaces() {
    this.logger.log('📋 Getting all pending places');
    return this.placesService.getAllPlaces('PENDING');
  }

  @Put('places/:id/activate')
  @ApiOperation({ summary: 'Activate a pending place' })
  @ApiResponse({ status: 200, type: PlaceDto })
  async activatePlace(@Param('id') id: string) {
    this.logger.log(`✅ Activating place ID: ${id}`);
    return this.placesService.updatePlaceStatus(Number(id), 'ACTIVE');
  }

  @Put('places/:id/reject')
  @ApiOperation({ summary: 'Reject a pending place (set to INACTIVE)' })
  @ApiResponse({ status: 200, type: PlaceDto })
  async rejectPlace(@Param('id') id: string) {
    this.logger.log(`❌ Rejecting place ID: ${id}`);
    return this.placesService.updatePlaceStatus(Number(id), 'INACTIVE');
  }
}
