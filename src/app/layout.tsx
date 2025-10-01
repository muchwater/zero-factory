import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import EnvChecker from '@/components/EnvChecker'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Zero Factory - 제로웨이스트 가이드',
  description: '제로웨이스트 라이프스타일을 위한 친환경 가이드 앱',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <body className={inter.className}>
        <EnvChecker>
          {children}
        </EnvChecker>
      </body>
    </html>
  )
}
