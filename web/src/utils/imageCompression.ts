/**
 * 이미지 파일을 중간 화질로 압축합니다
 * @param file 원본 이미지 파일
 * @param maxWidth 최대 너비 (기본값: 1200px)
 * @param quality 압축 품질 (0.0 ~ 1.0, 기본값: 0.7)
 * @returns 압축된 이미지 파일
 */
export async function compressImage(
  file: File,
  maxWidth: number = 1200,
  quality: number = 0.7
): Promise<File> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = (e) => {
      const img = new Image()

      img.onload = () => {
        // 캔버스 생성
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')

        if (!ctx) {
          reject(new Error('Canvas context not available'))
          return
        }

        // 이미지 크기 계산
        let width = img.width
        let height = img.height

        // 최대 너비를 초과하면 비율에 맞춰 축소
        if (width > maxWidth) {
          height = (height * maxWidth) / width
          width = maxWidth
        }

        // 캔버스 크기 설정
        canvas.width = width
        canvas.height = height

        // 이미지 그리기
        ctx.drawImage(img, 0, 0, width, height)

        // 압축된 이미지를 Blob으로 변환
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error('Failed to compress image'))
              return
            }

            // Blob을 File로 변환
            const compressedFile = new File(
              [blob],
              file.name,
              {
                type: 'image/jpeg',
                lastModified: Date.now(),
              }
            )

            console.log(`이미지 압축: ${(file.size / 1024).toFixed(1)}KB → ${(compressedFile.size / 1024).toFixed(1)}KB`)

            resolve(compressedFile)
          },
          'image/jpeg',
          quality
        )
      }

      img.onerror = () => {
        reject(new Error('Failed to load image'))
      }

      img.src = e.target?.result as string
    }

    reader.onerror = () => {
      reject(new Error('Failed to read file'))
    }

    reader.readAsDataURL(file)
  })
}
