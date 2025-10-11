import { ApiProperty } from '@nestjs/swagger';

export class CreateMemberDto {
  @ApiProperty({
    example: 'device-uuid-1234',
    description: '사용자 기기 ID (클라이언트에서 생성)',
  })
  deviceId: string;
}
