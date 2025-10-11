import { PrismaClient, PlaceCategory, PlaceType } from '@prisma/client';

const prisma = new PrismaClient();

function getRandomOffset() {
  // 위도/경도 1도 ≈ 111,000m
  // 100m ≈ 0.0009도
  const maxOffset = 0.0009;
  return (Math.random() - 0.5) * 2 * maxOffset;
}

async function createPlaceWithLocation(data, lng, lat) {
  const place = await prisma.place.create({ data });
  await prisma.$executeRawUnsafe(`
    UPDATE "Place"
    SET "location" = ST_SetSRID(ST_MakePoint(${lng}, ${lat}), 4326)::geography
    WHERE id = ${place.id};
  `);
  return place;
}

async function main() {
  // 데이터 초기화
  await prisma.storeOpeningHourException.deleteMany();
  await prisma.storeOpeningHour.deleteMany();
  await prisma.place.deleteMany();
  await prisma.member.deleteMany();

  const baseLng = 127.362;
  const baseLat = 36.3731;

  for (let i = 1; i <= 10; i++) {
    const lng = baseLng + getRandomOffset();
    const lat = baseLat + getRandomOffset();

    await createPlaceWithLocation(
      {
        name: `테스트 장소 ${i}`,
        description: i % 2 === 0 ? '리필/반납 가능' : '친환경 상품 판매',
        address: `대전광역시 유성구 테스트로 ${i}번길`,
        category: i % 2 === 0 ? PlaceCategory.FACILITY : PlaceCategory.STORE,
        types:
          i % 3 === 0
            ? [PlaceType.RETURN]
            : i % 3 === 1
              ? [PlaceType.RENT, PlaceType.BONUS]
              : [PlaceType.CLEAN],
        contact: i % 2 === 0 ? null : `010-0000-000${i}`,
        openingHours: {
          create: Array.from({ length: 7 }).map((_, day) => ({
            dayOfWeek: day,
            isClosed: day === 0, // 일요일 휴무
            openTime: day === 0 ? null : '09:00',
            closeTime: day === 0 ? null : '18:00',
          })),
        },
        exceptions:
          i === 1
            ? {
                create: [
                  { weekOfMonth: 2, dayOfWeek: 1, isClosed: true },
                  { date: new Date('2025-12-25'), openTime: '12:00', closeTime: '17:00' },
                ],
              }
            : undefined,
      },
      lng,
      lat,
    );
  }

  console.log('✅ 10 test places inserted within 100m');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(0);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
