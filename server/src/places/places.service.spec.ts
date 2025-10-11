import { Test, TestingModule } from '@nestjs/testing';
import { PlacesService } from './places.service';
import { PrismaService } from '../prisma/prisma.service';

describe('PlacesService', () => {
  let service: PlacesService;
  let prisma: PrismaService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        PlacesService,
        {
          provide: PrismaService,
          useValue: {
            place: {
              findMany: jest.fn().mockResolvedValue([
                {
                  id: 1,
                  name: '테스트 장소',
                  type: 'STORE',
                  address: '대전',
                },
              ]),
            },
            storeOpeningHour: { findFirst: jest.fn() },
            storeOpeningHourException: { findFirst: jest.fn() },
            $queryRawUnsafe: jest
              .fn()
              .mockResolvedValue([
                { id: 1, name: '근처 장소', type: 'STORE', address: '대전', distance: 42.3 },
              ]),
          },
        },
      ],
    }).compile();

    service = module.get<PlacesService>(PlacesService);
    prisma = module.get<PrismaService>(PrismaService);
  });

  it('서비스 정의됨', () => {
    expect(service).toBeDefined();
  });

  describe('getAllPlaces', () => {
    it('모든 장소를 store 포함해서 가져온다', async () => {
      const result = await service.getAllPlaces();
      expect(prisma.place.findMany).toHaveBeenCalledWith({
        include: { exceptions: true, openingHours: true },
      });
      expect(result[0].name).toBe('테스트 장소');
    });
  });

  describe('getPlacesNearby', () => {
    it('반경 내 장소를 가져온다', async () => {
      const result = await service.getPlacesNearby(36.3731, 127.362, 100);
      expect(prisma.$queryRawUnsafe).toHaveBeenCalled();
      expect(result[0]).toHaveProperty('distance');
      expect(result[0].distance).toBe(42.3);
    });
  });
});
