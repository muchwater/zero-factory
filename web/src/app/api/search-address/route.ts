import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const query = searchParams.get('query')

  if (!query) {
    return NextResponse.json(
      { error: '검색어를 입력해주세요.' },
      { status: 400 }
    )
  }

  // REST API 키 사용 (서버 사이드용)
  const apiKey = process.env.KAKAO_REST_API_KEY || process.env.NEXT_PUBLIC_KAKAO_MAP_KEY

  if (!apiKey) {
    console.error('카카오 API 키가 설정되지 않았습니다.')
    return NextResponse.json(
      { error: 'API 키가 설정되지 않았습니다.' },
      { status: 500 }
    )
  }

  console.log('카카오 API 호출:', query)
  console.log('API 키 사용 중:', apiKey.substring(0, 10) + '...')

  try {
    const response = await fetch(
      `https://dapi.kakao.com/v2/local/search/address.json?query=${encodeURIComponent(query)}`,
      {
        headers: {
          'Authorization': `KakaoAK ${apiKey}`
        }
      }
    )

    console.log('카카오 API 응답 상태:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('카카오 API 에러:', errorText)
      return NextResponse.json(
        { error: `카카오 API 오류: ${response.status}` },
        { status: response.status }
      )
    }

    const data = await response.json()
    console.log('검색 결과 갯수:', data.documents?.length || 0)
    return NextResponse.json(data)
  } catch (error) {
    console.error('주소 검색 오류:', error)
    return NextResponse.json(
      { error: '주소 검색 중 오류가 발생했습니다.' },
      { status: 500 }
    )
  }
}
