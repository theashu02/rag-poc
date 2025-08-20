import { authOption } from "@/lib/authOptions";
import NextAuth from "next-auth";


const handel = NextAuth(authOption);

export {handel as GET, handel as POST}