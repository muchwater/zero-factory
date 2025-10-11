import { Test, TestingModule } from '@nestjs/testing';
import { PlacesController } from './places.controller';
import { PlacesService } from './places.service';

describe('PlacesController', () => {
  let controller: PlacesController;
  let service: PlacesService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [PlacesController],
      providers: [
        {
          provide: PlacesService,
          useValue: {
            getAllPlaces: jest
              .fn()
              .mockResolvedValue([
                { id: 1, name: '테스트 장소', type: 'STORE', address: '대전', store: null },
              ]),
            getPlacesNearby: jest
              .fn()
              .mockResolvedValue([
                { id: 1, name: '근처 장소', type: 'FACILITY', address: '대전', distance: 42.1 },
              ]),
          },
        },
      ],
    }).compile();

    controller = module.get<PlacesController>(PlacesController);
    service = module.get<PlacesService>(PlacesService);
  });

  it('컨트롤러 정의됨', () => {
    expect(controller).toBeDefined();
  });

  describe('getAllPlaces', () => {
    it('모든 장소를 반환한다', async () => {
      const result = await controller.getAllPlaces();
      expect(service.getAllPlaces).toHaveBeenCalled();
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('테스트 장소');
    });
  });

  describe('getPlacesNearby', () => {
    it('근처 장소를 반환한다', async () => {
      const result = await controller.getPlacesNearby('36.3731', '127.3620', '100');
      expect(service.getPlacesNearby).toHaveBeenCalledWith(36.3731, 127.362, 100, 10, 0, undefined);
      expect(result).toHaveLength(1);
      expect(result[0]).toHaveProperty('distance', 42.1);
    });
  });
});
