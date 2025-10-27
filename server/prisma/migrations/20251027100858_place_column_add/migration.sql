-- CreateEnum
CREATE TYPE "public"."PlaceStateType" AS ENUM ('PENDING', 'ACTIVE', 'INACTIVE');

-- CreateEnum
CREATE TYPE "public"."BrandType" AS ENUM ('SUNHWA', 'UTURN');

-- AlterTable
ALTER TABLE "public"."Place" ADD COLUMN     "brand" "public"."BrandType",
ADD COLUMN     "state" "public"."PlaceStateType" NOT NULL DEFAULT 'PENDING';
