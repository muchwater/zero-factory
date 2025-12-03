// 귀여운 닉네임 생성 유틸리티

const ADJECTIVES = [
  '행복한',
  '귀여운',
  '용감한',
  '똑똑한',
  '활발한',
  '사랑스러운',
  '멋진',
  '씩씩한',
  '재미있는',
  '상냥한',
  '깜찍한',
  '명랑한',
  '든든한',
  '부지런한',
  '다정한',
]

const ANIMALS = [
  '판다',
  '코알라',
  '펭귄',
  '고양이',
  '강아지',
  '토끼',
  '여우',
  '다람쥐',
  '수달',
  '햄스터',
  '알파카',
  '돌고래',
  '사슴',
  '고래',
  '부엉이',
]

/**
 * 랜덤한 귀여운 닉네임을 생성합니다.
 * 형용사 + 동물 조합 (예: "행복한판다", "귀여운펭귄")
 * 총 225가지 조합 가능 (15 x 15)
 */
export function generateRandomNickname(): string {
  const randomAdjective = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)]
  const randomAnimal = ANIMALS[Math.floor(Math.random() * ANIMALS.length)]

  return `${randomAdjective}${randomAnimal}`
}
