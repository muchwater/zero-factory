import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsNotEmpty, IsOptional } from 'class-validator';

export class CreateMemberDto {
  @ApiProperty({
    example: '홍길동',
    description: '사용자 닉네임',
  })
  @IsString()
  @IsNotEmpty()
  nickname: string;

  @ApiProperty({
    example: 'device-uuid-1234',
    description: '사용자 기기 ID (클라이언트에서 생성)',
    required: false,
  })
  @IsString()
  @IsOptional()
  deviceId?: string;
}
