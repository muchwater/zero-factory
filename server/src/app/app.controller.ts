import { Controller, Get } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { PrismaService } from '../prisma/prisma.service';
import { HealthResponseDto } from './dto/health-response.dto';

@ApiTags('health')
@Controller()
export class AppController {
  constructor(private readonly prisma: PrismaService) {}

  @Get('ping')
  @ApiOperation({ summary: 'Ping / Pong Health Check' })
  ping() {
    return { message: 'pong' };
  }

  @Get('health')
  @ApiOperation({ summary: '서비스 헬스체크', description: '서버/DB 상태를 확인합니다.' })
  @ApiResponse({ status: 200, type: HealthResponseDto })
  async health(): Promise<HealthResponseDto> {
    let dbStatus = 'ok';
    try {
      // 간단한 쿼리 실행으로 DB 연결 체크
      await this.prisma.$queryRawUnsafe(`SELECT 1`);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (e) {
      dbStatus = 'error';
    }

    return {
      status: 'ok',
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
      database: dbStatus,
    };
  }
}
