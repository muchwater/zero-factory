import { Test, TestingModule } from '@nestjs/testing';
import { MembersController } from './members.controller';
import { MembersService } from './members.service';

describe('MembersController', () => {
  let controller: MembersController;
  let service: MembersService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [MembersController],
      providers: [
        {
          provide: MembersService,
          useValue: {
            findOrCreate: jest.fn().mockResolvedValue({
              id: 'uuid-1234',
              deviceId: 'test-device',
              pointBalance: 0,
              createdAt: new Date(),
            }),
          },
        },
      ],
    }).compile();

    controller = module.get<MembersController>(MembersController);
    service = module.get<MembersService>(MembersService);
  });

  it('컨트롤러 정의됨', () => {
    expect(controller).toBeDefined();
  });

  it('findOrCreate 호출 시 멤버를 반환한다', async () => {
    const result = await controller.findOrCreate({ deviceId: 'test-device' });
    expect(result.deviceId).toBe('test-device');
    expect(service.findOrCreate).toHaveBeenCalledWith('test-device');
  });
});
