import { ApiProperty } from '@nestjs/swagger';

export class StoreOpeningHourDto {
  @ApiProperty({ description: '고유 ID', example: 1 })
  id: number;

  @ApiProperty({ description: '요일 (0=일요일, 6=토요일)', example: 1 })
  dayOfWeek: number;

  @ApiProperty({ description: '휴무 여부', example: false })
  isClosed: boolean;

  @ApiProperty({ description: '개점 시간 (HH:mm)', example: '09:00', required: false })
  openTime?: string | null;

  @ApiProperty({ description: '폐점 시간 (HH:mm)', example: '18:00', required: false })
  closeTime?: string | null;
}
