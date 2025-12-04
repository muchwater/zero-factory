-- CreateEnum
CREATE TYPE "public"."ReceiptStatus" AS ENUM ('PENDING', 'APPROVED', 'REJECTED');

-- DropForeignKey
ALTER TABLE "public"."PointTransaction" DROP CONSTRAINT "PointTransaction_placeId_fkey";

-- AlterTable
ALTER TABLE "public"."Member" ADD COLUMN     "lastReceiptAt" TIMESTAMP(3),
ADD COLUMN     "receiptRestricted" BOOLEAN NOT NULL DEFAULT false;

-- AlterTable
ALTER TABLE "public"."PointTransaction" ALTER COLUMN "placeId" DROP NOT NULL;

-- CreateTable
CREATE TABLE "public"."Receipt" (
    "id" SERIAL NOT NULL,
    "memberId" TEXT NOT NULL,
    "placeId" INTEGER,
    "productDescription" TEXT NOT NULL,
    "photoPath" TEXT NOT NULL,
    "pointsEarned" INTEGER NOT NULL DEFAULT 100,
    "status" "public"."ReceiptStatus" NOT NULL DEFAULT 'PENDING',
    "verificationResult" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Receipt_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Receipt_memberId_idx" ON "public"."Receipt"("memberId");

-- CreateIndex
CREATE INDEX "Receipt_status_idx" ON "public"."Receipt"("status");

-- CreateIndex
CREATE INDEX "Receipt_createdAt_idx" ON "public"."Receipt"("createdAt");

-- AddForeignKey
ALTER TABLE "public"."PointTransaction" ADD CONSTRAINT "PointTransaction_placeId_fkey" FOREIGN KEY ("placeId") REFERENCES "public"."Place"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."Receipt" ADD CONSTRAINT "Receipt_memberId_fkey" FOREIGN KEY ("memberId") REFERENCES "public"."Member"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."Receipt" ADD CONSTRAINT "Receipt_placeId_fkey" FOREIGN KEY ("placeId") REFERENCES "public"."Place"("id") ON DELETE SET NULL ON UPDATE CASCADE;
