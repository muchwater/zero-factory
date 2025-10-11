import { Test, TestingModule } from '@nestjs/testing';
import { AppController } from './app.controller';
import { PrismaService } from '../prisma/prisma.service';

describe('AppController', () => {
  let controller: AppController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [AppController],
      providers: [PrismaService],
    }).compile();

    controller = module.get<AppController>(AppController);
  });

  it('컨트롤러 정의됨', () => {
    expect(controller).toBeDefined();
  });

  it('ping 호출 시 pong 반환', () => {
    expect(controller.ping()).toEqual({ message: 'pong' });
  });

  it('health 호출 시 상태 반환', async () => {
    const result = await controller.health();
    expect(result).toHaveProperty('status', 'ok');
    expect(result).toHaveProperty('database');
  });
});
