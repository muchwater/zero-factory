// 장소 데이터를 관리하는 커스텀 훅

import { useState, useEffect, useCallback } from 'react'
import { placesApi } from '@/services/api'
import type { Place, PlaceNearby } from '@/types/api'

interface UsePlacesOptions {
  lat?: number
  lng?: number
  radius?: number
  autoFetch?: boolean
}

export function usePlaces(options: UsePlacesOptions = {}) {
  const [places, setPlaces] = useState<Place[]>([])
  const [nearbyPlaces, setNearbyPlaces] = useState<PlaceNearby[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 전체 장소 조회
  const fetchAllPlaces = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const data = await placesApi.getAllPlaces()
      setPlaces(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : '장소 정보를 불러오는데 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }, [])

  // 근처 장소 검색
  const fetchNearbyPlaces = useCallback(async (params: {
    lat: number
    lng: number
    radius: number
    limit?: number
    offset?: number
    types?: string[]
  }) => {
    setLoading(true)
    setError(null)
    
    try {
      const data = await placesApi.getPlacesNearby(params)
      setNearbyPlaces(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : '근처 장소를 검색하는데 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }, [])

  // 특정 장소 조회
  const fetchPlaceById = useCallback(async (id: number) => {
    setLoading(true)
    setError(null)
    
    try {
      const data = await placesApi.getPlaceById(id)
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : '장소 정보를 불러오는데 실패했습니다.')
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  // 자동으로 데이터 가져오기
  useEffect(() => {
    if (options.autoFetch) {
      if (options.lat && options.lng && options.radius) {
        fetchNearbyPlaces({
          lat: options.lat,
          lng: options.lng,
          radius: options.radius,
          limit: 20
        })
      } else {
        fetchAllPlaces()
      }
    }
  }, [options.autoFetch, options.lat, options.lng, options.radius, fetchAllPlaces, fetchNearbyPlaces])

  return {
    places,
    nearbyPlaces,
    loading,
    error,
    fetchAllPlaces,
    fetchNearbyPlaces,
    fetchPlaceById,
    refetch: options.lat && options.lng && options.radius ? 
      () => fetchNearbyPlaces({
        lat: options.lat!,
        lng: options.lng!,
        radius: options.radius!,
        limit: 20
      }) : 
      fetchAllPlaces
  }
}

