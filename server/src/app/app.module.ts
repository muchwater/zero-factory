import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { ConfigModule } from '@nestjs/config';
import { ServeStaticModule } from '@nestjs/serve-static';
import { join } from 'path';
import { PlacesModule } from '../places/places.module';
import { MembersModule } from '../members/members.module';
import { PrismaModule } from '../prisma/prisma.module';
import { AdminModule } from '../admin/admin.module';
import { PointsModule } from '../points/points.module';
import { ReceiptsModule } from '../receipts/receipts.module';

@Module({
  imports: [
    ConfigModule,
    ServeStaticModule.forRoot({
      rootPath: join(process.cwd(), 'uploads'),
      serveRoot: '/uploads',
    }),
    PlacesModule,
    MembersModule,
    PrismaModule,
    AdminModule,
    PointsModule,
    ReceiptsModule,
  ],
  controllers: [AppController],
  providers: [],
})
export class AppModule {}
