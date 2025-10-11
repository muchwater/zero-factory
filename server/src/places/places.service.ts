import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { PlaceNearbyDto } from './dto/place-nearby.dto';
import { getDayName, getWeekOfMonth } from '../common/day';

@Injectable()
export class PlacesService {
  constructor(private prisma: PrismaService) {}

  async getAllPlaces() {
    return this.prisma.place.findMany({
      include: {
        openingHours: true,
        exceptions: true,
      },
    });
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
    const places =
      await this.prisma.$queryRawUnsafe<Array<Omit<PlaceNearbyDto, 'todayHours'>>>(query);

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

      results.push({ ...p, todayHours });
    }

    return results;
  }
}
