'use client'

import React, { createContext, useState, useEffect, useCallback } from 'react'
import type { Member } from '@/types/api'
import { membersApi, ApiError } from '@/services/api'
import { getMemberId, setMemberId, removeMemberId } from '@/utils/cookies'
import { generateRandomNickname } from '@/utils/nickname'

interface MemberContextValue {
  member: Member | null
  loading: boolean
  error: Error | null
  refreshMember: () => Promise<void>
  updateNickname: (nickname: string) => Promise<void>
}

export const MemberContext = createContext<MemberContextValue | undefined>(undefined)

export function MemberProvider({ children }: { children: React.ReactNode }) {
  const [member, setMember] = useState<Member | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [initializing, setInitializing] = useState(false)

  const createAnonymousMember = useCallback(async () => {
    try {
      const nickname = generateRandomNickname()
      const deviceId = crypto.randomUUID()

      const newMember = await membersApi.findOrCreate({
        nickname,
        deviceId,
      })

      setMemberId(newMember.id)
      setMember(newMember)
      setError(null)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create member'
      setError(new Error(errorMessage))
      throw err
    }
  }, [])

  const initializeMember = useCallback(async () => {
    if (initializing) return

    setInitializing(true)
    setLoading(true)
    setError(null)

    try {
      const memberId = getMemberId()

      if (memberId) {
        try {
          const existingMember = await membersApi.getById(memberId)
          setMember(existingMember)
        } catch (err) {
          if (err instanceof ApiError && err.status === 404) {
            removeMemberId()
            await createAnonymousMember()
          } else {
            throw err
          }
        }
      } else {
        await createAnonymousMember()
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to initialize member'
      setError(new Error(errorMessage))
    } finally {
      setLoading(false)
      setInitializing(false)
    }
  }, [initializing, createAnonymousMember])

  const refreshMember = useCallback(async () => {
    const memberId = getMemberId()
    if (!memberId) return

    try {
      const updatedMember = await membersApi.getById(memberId)
      setMember(updatedMember)
      setError(null)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to refresh member'
      setError(new Error(errorMessage))
    }
  }, [])

  const updateNickname = useCallback(async (nickname: string) => {
    const memberId = getMemberId()
    if (!memberId) {
      throw new Error('No member ID found')
    }

    try {
      const updatedMember = await membersApi.findOrCreate({
        nickname,
        deviceId: member?.deviceId || undefined,
      })
      setMember(updatedMember)
      setError(null)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update nickname'
      setError(new Error(errorMessage))
      throw err
    }
  }, [member?.deviceId])

  useEffect(() => {
    initializeMember()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const value: MemberContextValue = {
    member,
    loading,
    error,
    refreshMember,
    updateNickname,
  }

  return <MemberContext.Provider value={value}>{children}</MemberContext.Provider>
}
