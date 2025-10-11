import { ApiProperty } from '@nestjs/swagger';

export class StoreOpeningHourExceptionDto {
  @ApiProperty({ description: '고유 ID', example: 1 })
  id: number;

  @ApiProperty({ description: '몇 번째 주 (1=첫째주, null이면 무시)', example: 2, required: false })
  weekOfMonth?: number | null;

  @ApiProperty({
    description: '요일 (0=일요일, 6=토요일, null이면 무시)',
    example: 1,
    required: false,
  })
  dayOfWeek?: number | null;

  @ApiProperty({
    description: '특정 날짜 (예외적 운영일자)',
    example: '2025-08-15',
    required: false,
  })
  date?: Date | null;

  @ApiProperty({ description: '휴무 여부', example: true })
  isClosed: boolean;

  @ApiProperty({ description: '개점 시간 (예외)', example: '12:00', required: false })
  openTime?: string | null;

  @ApiProperty({ description: '폐점 시간 (예외)', example: '17:00', required: false })
  closeTime?: string | null;
}
