/*
  Warnings:

  - A unique constraint covering the columns `[nickname]` on the table `Member` will be added. If there are existing duplicate values, this will fail.
  - Made the column `nickname` on table `Member` required. This step will fail if there are existing NULL values in that column.

*/
-- AlterTable
ALTER TABLE "public"."Member" ALTER COLUMN "nickname" SET NOT NULL,
ALTER COLUMN "deviceId" DROP NOT NULL;

-- CreateIndex
CREATE UNIQUE INDEX "Member_nickname_key" ON "public"."Member"("nickname");
