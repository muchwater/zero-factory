import { NestFactory } from '@nestjs/core';
import { AppModule } from './app/app.module';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { INestApplication, ValidationPipe } from '@nestjs/common';

export async function createApp(): Promise<INestApplication> {
  const app = await NestFactory.create(AppModule);

  // ValidationPipe 활성화 - DTO 변환 및 검증
  app.useGlobalPipes(
    new ValidationPipe({
      transform: true, // DTO 클래스로 자동 변환
      whitelist: true, // DTO에 없는 속성 제거
      forbidNonWhitelisted: false, // 추가 속성이 있어도 오류 발생하지 않음
    }),
  );

  // CORS 활성화
  app.enableCors({
    origin: [
      'http://localhost:5000',
      'http://localhost:3001',
      'http://43.201.190.116:3001',
      'https://zeromap.store',
      'https://www.zeromap.store',
      'http://zeromap.store',
      'http://www.zeromap.store'
    ],
    credentials: true,
  });

  const config = new DocumentBuilder()
    .setTitle('ZeroWaste API')
    .setDescription('제로웨이스트 앱용 백엔드 API 문서')
    .setVersion('1.0')
    .addTag('members')
    .addTag('places')
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api-docs', app, document);

  return app;
}

async function bootstrap(): Promise<void> {
  const app = await createApp();
  await app.listen(3000).then(() => console.log('http://localhost:3000/api-docs'));
}

bootstrap();
