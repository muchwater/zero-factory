import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { ConfigModule } from '@nestjs/config';
import { PlacesModule } from '../places/places.module';
import { MembersModule } from '../members/members.module';
import { PrismaModule } from '../prisma/prisma.module';
@Module({
  imports: [ConfigModule, PlacesModule, MembersModule, PrismaModule],
  controllers: [AppController],
  providers: [],
})
export class AppModule {}
