import { Test, TestingModule } from '@nestjs/testing';
import { MembersService } from './members.service';
import { PrismaService } from '../prisma/prisma.service';

describe('MembersService', () => {
  let service: MembersService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [MembersService, PrismaService],
    }).compile();

    service = module.get<MembersService>(MembersService);
  });

  it('서비스 정의됨', () => {
    expect(service).toBeDefined();
  });

  it('새 멤버를 생성한다', async () => {
    const deviceId = 'no-device-id';
    const member = await service.findOrCreate(deviceId);
    expect(member.deviceId).toBe(deviceId);
    expect(member.pointBalance).toBe(0);
  });

  it('기존 멤버를 찾는다', async () => {
    const deviceId = 'test-device-id';
    const member = await service.findOrCreate(deviceId);
    expect(member.deviceId).toBe(deviceId);
    expect(member.pointBalance).toBe(0);
  });
});
