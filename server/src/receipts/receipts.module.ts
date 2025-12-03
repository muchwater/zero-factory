import { Module } from '@nestjs/common';
import { ReceiptsController } from './receipts.controller';
import { ReceiptsService } from './receipts.service';
import { PrismaModule } from '../prisma/prisma.module';
import { PointsModule } from '../points/points.module';

@Module({
  imports: [PrismaModule, PointsModule],
  controllers: [ReceiptsController],
  providers: [ReceiptsService],
})
export class ReceiptsModule {}
