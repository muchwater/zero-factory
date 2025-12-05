import { PrismaClient, PlaceCategory, PlaceType } from '@prisma/client';

const prisma = new PrismaClient();

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

  // 실제 유턴컵(다회용컵) 대여 가능 카페 데이터
  const realCafes = [
    {
      name: '커피 볶는 집',
      description: '유턴컵 (다회용컵) 대여 가능',
      address: '대전광역시 유성구',
      lat: 36.363414300,
      lng: 127.357935057,
      contact: '042-XXX-XXXX',
    },
    {
      name: '노은도서관',
      description: '유턴컵 (다회용컵) 대여 가능',
      address: '대전광역시 유성구 노은동',
      lat: 36.381493476,
      lng: 127.320662759,
      contact: null,
    },
    {
      name: '유성구청',
      description: '유턴컵 (다회용컵) 대여 가능',
      address: '대전광역시 유성구 대학로 211',
      lat: 36.362218923,
      lng: 127.356148463,
      contact: '042-611-5114',
    },
    {
      name: '커피바 유성별',
      description: '유턴컵 (다회용컵) 대여 가능',
      address: '대전광역시 유성구',
      lat: 36.361992296,
      lng: 127.354850895,
      contact: '042-XXX-XXXX',
    },
  ];

  // 각 카페를 데이터베이스에 추가
  for (const cafe of realCafes) {
    await createPlaceWithLocation(
      {
        name: cafe.name,
        description: cafe.description,
        address: cafe.address,
        category: PlaceCategory.STORE,
        types: [PlaceType.RENT], // 대여 + 포인트 적립
        contact: cafe.contact,
        state: "ACTIVE",
        openingHours: {
          create: Array.from({ length: 7 }).map((_, day) => ({
            dayOfWeek: day,
            isClosed: day === 0, // 일요일 휴무
            openTime: day === 0 ? null : '09:00',
            closeTime: day === 0 ? null : '21:00', // 카페는 21시까지
          })),
        },
      },
      cafe.lng,
      cafe.lat,
    );
  }

  console.log(`✅ ${realCafes.length}개의 실제 유턴컵 대여 카페가 추가되었습니다.`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(0);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
