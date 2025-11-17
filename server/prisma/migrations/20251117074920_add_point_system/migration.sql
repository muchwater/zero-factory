-- CreateTable
CREATE TABLE "public"."PointLog" (
    "id" SERIAL NOT NULL,
    "memberId" TEXT NOT NULL,
    "sumBalance" INTEGER NOT NULL,
    "yearMonth" TEXT NOT NULL,
    "earnedPoints" INTEGER NOT NULL DEFAULT 0,
    "redeemedPoints" INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT "PointLog_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "public"."PointLog" ADD CONSTRAINT "PointLog_memberId_fkey" FOREIGN KEY ("memberId") REFERENCES "public"."Member"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
