import { ApiProperty } from '@nestjs/swagger';

export class MemberResponseDto {
  @ApiProperty({
    example: 'f3b5c8f0-9c2a-4c58-91d1-87db16c6c7e4',
    description: 'Member UUID',
  })
  id: string;

  @ApiProperty({
    example: 'device-uuid-1234',
    description: '클라이언트에서 전달한 기기 식별자',
  })
  deviceId: string;

  @ApiProperty({
    example: 0,
    description: '현재 포인트 잔액',
  })
  pointBalance: number;

  @ApiProperty({
    example: '2025-09-20T12:34:56.000Z',
    description: '멤버 생성 시각',
  })
  createdAt: Date;
}
