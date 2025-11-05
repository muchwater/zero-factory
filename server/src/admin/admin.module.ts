import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AdminController } from './admin.controller';
import { PlacesModule } from '../places/places.module';
import { AdminCodeGuard } from './admin-code.guard';

@Module({
  imports: [ConfigModule, PlacesModule],
  controllers: [AdminController],
  providers: [AdminCodeGuard],
})
export class AdminModule {}
