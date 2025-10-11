import { Controller, Post, Body } from '@nestjs/common';
import { MembersService } from './members.service';
import { CreateMemberDto } from './dto/create-member.dto';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { MemberResponseDto } from './dto/member-response.dto';

@ApiTags('members')
@Controller('members')
export class MembersController {
  constructor(private readonly membersService: MembersService) {}

  @Post()
  @ApiOperation({
    summary: '멤버 생성/조회',
    description: 'deviceId로 멤버를 찾거나 없으면 새로 생성합니다.',
  })
  @ApiResponse({ status: 201, description: '멤버가 생성되거나 조회됨.', type: MemberResponseDto })
  async findOrCreate(@Body() dto: CreateMemberDto): Promise<MemberResponseDto> {
    return this.membersService.findOrCreate(dto.deviceId);
  }
}
