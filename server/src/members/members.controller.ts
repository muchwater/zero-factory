import { Controller, Post, Body, Get, Param } from '@nestjs/common';
import { MembersService } from './members.service';
import { CreateMemberDto } from './dto/create-member.dto';
import { ApiTags, ApiOperation, ApiResponse, ApiParam } from '@nestjs/swagger';
import { MemberResponseDto } from './dto/member-response.dto';

@ApiTags('members')
@Controller('members')
export class MembersController {
  constructor(private readonly membersService: MembersService) {}

  @Post()
  @ApiOperation({
    summary: '멤버 생성/조회',
    description: 'nickname으로 멤버를 찾거나 없으면 새로 생성합니다.',
  })
  @ApiResponse({ status: 201, description: '멤버가 생성되거나 조회됨.', type: MemberResponseDto })
  async findOrCreate(@Body() dto: CreateMemberDto): Promise<MemberResponseDto> {
    return this.membersService.findOrCreate(dto.nickname, dto.deviceId);
  }

  @Get(':memberId')
  @ApiOperation({
    summary: '멤버 조회',
    description: 'memberId로 특정 멤버의 정보를 조회합니다.',
  })
  @ApiParam({
    name: 'memberId',
    description: '조회할 멤버의 UUID',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  @ApiResponse({ status: 200, description: '멤버 정보 조회 성공.', type: MemberResponseDto })
  @ApiResponse({ status: 404, description: '멤버를 찾을 수 없음.' })
  async findById(@Param('memberId') memberId: string): Promise<MemberResponseDto> {
    return this.membersService.findById(memberId);
  }
}
