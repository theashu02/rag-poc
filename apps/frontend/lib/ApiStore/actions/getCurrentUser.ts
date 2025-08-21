'use server'

import { getServerSession } from 'next-auth';
import User from '@/Models/User';
import connectDB from '@/lib/mongodb';
import { authOption } from '@/lib/authOptions';

export async function getUser() {
  const session = await getServerSession(authOption);
  if (!session?.user?.email) return null;
  await connectDB();
  const user = await User.findOne({ email: session.user.email });
  if (!user) return null;
  return {
    userId:   user._id.toString(),
    email:    user.email,
    name:     user.name
  };
}