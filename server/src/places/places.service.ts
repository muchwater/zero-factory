import { Injectable, Logger } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { PlaceNearbyDto } from './dto/place-nearby.dto';
import { getDayName, getWeekOfMonth } from '../common/day';

@Injectable()
export class PlacesService {
  private readonly logger = new Logger(PlacesService.name);
  
  constructor(private prisma: PrismaService) {}

  async getAllPlaces() {
    this.logger.log('📚 Getting all places...');
    
    const places = await this.prisma.place.findMany({
      include: {
        openingHours: true,
        exceptions: true,
      },
    });
    
    this.logger.log(`✅ Found ${places.length} total places`);
    
    // location 정보 추가
    const placesWithLocation = await Promise.all(
      places.map(async (place) => {
        const result = await this.prisma.$queryRawUnsafe<Array<{ lat: number; lng: number }>>(
          `SELECT ST_Y(location::geometry) AS lat, ST_X(location::geometry) AS lng FROM "Place" WHERE id = ${place.id}`
        );
        
        const location = result[0] ? { lat: result[0].lat, lng: result[0].lng } : undefined;
        
        return {
          ...place,
          location,
        };
      })
    );
    
    return placesWithLocation;
  }

  async getPlaceById(id: number) {
    return this.prisma.place.findUnique({
      where: { id },
      include: {
        openingHours: true,
        exceptions: true,
      },
    });
  }

  async getPlacesNearby(
    lat: number,
    lng: number,
    radiusMeters: number,
    limit = 10,
    offset = 0,
    filterTypes?: string[],
  ): Promise<PlaceNearbyDto[]> {
    const query = `
  SELECT 
    id, 
    name, 
    description, 
    address, 
    category, 
    types, 
    contact,
    ST_Y(location::geometry) AS lat,
    ST_X(location::geometry) AS lng,
    ROUND(
      ST_Distance(
        ST_MakePoint(${lng}, ${lat})::geography,
        "location"
      )::numeric, 1
    )::float8 AS distance
  FROM "Place"
  WHERE ST_DWithin(
    "location",
    ST_MakePoint(${lng}, ${lat})::geography,
    ${radiusMeters}
  )
  ${
    filterTypes && filterTypes.length > 0
      ? `AND types && ARRAY[${filterTypes.map((t) => `'${t.toUpperCase()}'`).join(',')}]::"PlaceType"[]`
      : ''
  }
  ORDER BY distance ASC
  LIMIT ${limit}
  OFFSET ${offset};
`;
    this.logger.log(`🔍 Searching places: lat=${lat}, lng=${lng}, radius=${radiusMeters}m`);
    
    const places =
      await this.prisma.$queryRawUnsafe<Array<Omit<PlaceNearbyDto, 'todayHours'>>>(query);
    
    this.logger.log(`📊 Database returned ${places.length} places`);
    if (places.length > 0) {
      this.logger.log(`First place: ${places[0].name} at (${places[0].lat}, ${places[0].lng})`);
    }

    const today = new Date();
    const todayDay = today.getDay();
    const todayWeek = getWeekOfMonth(today);
    const dayName = getDayName(todayDay);

    const results: PlaceNearbyDto[] = [];

    for (const p of places) {
      const openingHours = await this.prisma.storeOpeningHour.findFirst({
        where: { placeId: p.id, dayOfWeek: todayDay },
      });

      let todayHours = openingHours
        ? {
            isClosed: openingHours.isClosed,
            openTime: openingHours.openTime,
            closeTime: openingHours.closeTime,
            dayName,
          }
        : { isClosed: true };

      const exception = await this.prisma.storeOpeningHourException.findFirst({
        where: {
          placeId: p.id,
          OR: [{ date: today }, { weekOfMonth: todayWeek, dayOfWeek: todayDay }],
        },
      });

      if (exception) {
        todayHours = {
          isClosed: exception.isClosed,
          openTime: exception.openTime,
          closeTime: exception.closeTime,
          dayName,
        };
      }

      results.push({ 
        ...p, 
        location: p.lat && p.lng ? { lat: p.lat, lng: p.lng } : undefined,
        todayHours 
      });
    }

    return results;
  }
}
