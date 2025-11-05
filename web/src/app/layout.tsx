import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Script from 'next/script'
import './globals.css'
import EnvChecker from '@/components/EnvChecker'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Zero Factory - 제로웨이스트 가이드',
  description: '제로웨이스트 라이프스타일을 위한 친환경 가이드 앱',
  icons: {
    icon: '/favicon.ico',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <body className={inter.className}>
        <Script
          src={`//dapi.kakao.com/v2/maps/sdk.js?appkey=${process.env.NEXT_PUBLIC_KAKAO_MAP_KEY}&libraries=services&autoload=false`}
          strategy="beforeInteractive"
        />
        <EnvChecker>
          {children}
        </EnvChecker>
      </body>
    </html>
  )
}
