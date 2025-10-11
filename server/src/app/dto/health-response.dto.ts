import { ApiProperty } from '@nestjs/swagger';

export class HealthResponseDto {
  @ApiProperty({ example: 'ok', description: '서비스 상태' })
  status: string;

  @ApiProperty({ example: 123.45, description: '서버 업타임 (초)' })
  uptime: number;

  @ApiProperty({ example: '2025-09-20T15:00:00.000Z', description: '체크 시각' })
  timestamp: string;

  @ApiProperty({ example: 'ok', description: '데이터베이스 연결 상태' })
  database: string;
}
