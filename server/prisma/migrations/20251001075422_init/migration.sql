-- CreateExtension
CREATE EXTENSION IF NOT EXISTS "postgis";

-- CreateEnum
CREATE TYPE "public"."PlaceCategory" AS ENUM ('STORE', 'FACILITY');

-- CreateEnum
CREATE TYPE "public"."PlaceType" AS ENUM ('RENT', 'RETURN', 'BONUS', 'CLEAN');

-- CreateEnum
CREATE TYPE "public"."TransactionType" AS ENUM ('EARN', 'REDEEM');

-- CreateTable
CREATE TABLE "public"."Place" (
    "id" SERIAL NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "address" TEXT NOT NULL,
    "category" "public"."PlaceCategory" NOT NULL,
    "location" geography(Point,4326),
    "types" "public"."PlaceType"[],
    "contact" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Place_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."StoreOpeningHour" (
    "id" SERIAL NOT NULL,
    "placeId" INTEGER NOT NULL,
    "dayOfWeek" INTEGER NOT NULL,
    "isClosed" BOOLEAN NOT NULL DEFAULT false,
    "openTime" TEXT,
    "closeTime" TEXT,

    CONSTRAINT "StoreOpeningHour_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."StoreOpeningHourException" (
    "id" SERIAL NOT NULL,
    "placeId" INTEGER NOT NULL,
    "weekOfMonth" INTEGER,
    "dayOfWeek" INTEGER,
    "date" TIMESTAMP(3),
    "isClosed" BOOLEAN NOT NULL DEFAULT false,
    "openTime" TEXT,
    "closeTime" TEXT,

    CONSTRAINT "StoreOpeningHourException_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."Member" (
    "id" TEXT NOT NULL,
    "nickname" TEXT,
    "deviceId" TEXT NOT NULL,
    "pointBalance" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Member_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."PointTransaction" (
    "id" SERIAL NOT NULL,
    "memberId" TEXT NOT NULL,
    "placeId" INTEGER NOT NULL,
    "amount" INTEGER NOT NULL,
    "type" "public"."TransactionType" NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PointTransaction_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Member_deviceId_key" ON "public"."Member"("deviceId");

-- AddForeignKey
ALTER TABLE "public"."StoreOpeningHour" ADD CONSTRAINT "StoreOpeningHour_placeId_fkey" FOREIGN KEY ("placeId") REFERENCES "public"."Place"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."StoreOpeningHourException" ADD CONSTRAINT "StoreOpeningHourException_placeId_fkey" FOREIGN KEY ("placeId") REFERENCES "public"."Place"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."PointTransaction" ADD CONSTRAINT "PointTransaction_memberId_fkey" FOREIGN KEY ("memberId") REFERENCES "public"."Member"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."PointTransaction" ADD CONSTRAINT "PointTransaction_placeId_fkey" FOREIGN KEY ("placeId") REFERENCES "public"."Place"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
