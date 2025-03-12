"use client";

import { LoadingSVG } from "@/components/button/LoadingSVG";
import { ChatMessageType, ChatTile } from "@/components/chat/ChatTile";
import { AudioInputTile } from "@/components/config/AudioInputTile";
import { ConfigurationPanelItem } from "@/components/config/ConfigurationPanelItem";
import { NameValueRow } from "@/components/config/NameValueRow";
import { PlaygroundHeader } from "@/components/playground/PlaygroundHeader";
import { PlaygroundTile } from "@/components/playground/PlaygroundTile";
import { useConfig } from "@/hooks/useConfig";
import { useMultibandTrackVolume } from "@/hooks/useTrackVolume";
import { AgentState } from "@/lib/types";
import {
  VideoTrack,
  useChat,
  useConnectionState,
  useDataChannel,
  useLocalParticipant,
  useRemoteParticipants,
  useRoomInfo,
  useTracks,
} from "@livekit/components-react";
import {
  ConnectionState,
  LocalParticipant,
  RoomEvent,
  Track,
} from "livekit-client";
import { ReactNode, useCallback, useEffect, useMemo, useState } from 'react'

export interface PlaygroundMeta {
  name: string;
  value: string;
}

export interface PlaygroundProps {
  logo?: ReactNode;
  themeColors: string[];
  onConnect: (connect: boolean, opts?: { token: string; url: string }) => void;
}

const headerHeight = 56;

export default function Playground({
  logo,
  themeColors,
  onConnect,
}: PlaygroundProps) {
  const { config, setUserSettings } = useConfig();
  const { name } = useRoomInfo();
  const [agentState, setAgentState] = useState<AgentState>("offline");
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [transcripts, setTranscripts] = useState<ChatMessageType[]>([]);
  const { localParticipant } = useLocalParticipant();

  const participants = useRemoteParticipants({
    updateOnlyOn: [RoomEvent.ParticipantMetadataChanged],
  });
  const agentParticipant = participants.find((p) => p.isAgent);

  const { send: sendChat, chatMessages } = useChat();
  const roomState = useConnectionState();
  const tracks = useTracks();

  const isAgentConnected = agentState !== "offline";

  const tabs = ['视频', '音频', '文本'];
  const [activeTab, setActiveTab] = useState(0);
  const isTabActive = useMemo(() => {
    return roomState === ConnectionState.Disconnected && !isAgentConnected;
  }, [roomState, isAgentConnected]);
  const needMicrophone = useMemo(() => {
    return activeTab === 0 || activeTab === 1;
  }, [activeTab]);

  useEffect(() => {
    if (roomState === ConnectionState.Connected && needMicrophone) {
      try {
        localParticipant.setMicrophoneEnabled(config.settings.inputs.mic);
        if (activeTab === 0) {
          localParticipant.setCameraEnabled(config.settings.inputs.camera);
        }
      } catch (error) {
        console.error("Failed to enable media devices:", error);
      }
    }
  }, [config, localParticipant, roomState, activeTab]);

  const localTracks = tracks.filter(
    ({ participant }) => participant instanceof LocalParticipant
  );
  const localVideoTrack = localTracks.find(
    ({ source }) => source === Track.Source.Camera
  );
  const localMicTrack = localTracks.find(
    ({ source }) => source === Track.Source.Microphone
  );

  const localMultibandVolume = useMultibandTrackVolume(
    localMicTrack?.publication.track,
    20
  );

  const shouldShowChat = useMemo(() => {
    return activeTab === 2 && isAgentConnected;
  }, [activeTab, isAgentConnected]);

  useEffect(() => {
    if (!agentParticipant) {
      setAgentState("offline");
      return;
    }
    let agentMd: any = {};
    if (agentParticipant.metadata) {
      try {
        agentMd = JSON.parse(agentParticipant.metadata);
      } catch (error) {
        console.error("Failed to parse agent metadata:", error);
        return;
      }
    }
    if (agentMd.agent_state) {
      setAgentState(agentMd.agent_state);
    } else {
      setAgentState("starting");
    }
  }, [agentParticipant, agentParticipant?.metadata]);

  const onDataReceived = useCallback((msg: any) => {
    if (msg.topic === "transcription") {
      const decoded = JSON.parse(new TextDecoder("utf-8").decode(msg.payload))
      let timestamp = new Date().getTime();
      if ("timestamp" in decoded && decoded.timestamp > 0) {
        timestamp = decoded.timestamp;
      }
      setTranscripts([
        ...transcripts,
        {
          name: "You",
          message: decoded.text,
          timestamp: timestamp,
          isSelf: true,
        },
      ]);
    }
  }, [transcripts]);

  // combine transcripts and chat together
  useEffect(() => {
    const allMessages = [...transcripts];
    for (const msg of chatMessages) {
      const isAgent = msg.from?.identity === agentParticipant?.identity;
      const isSelf = msg.from?.identity === localParticipant?.identity;
      let name = msg.from?.name;
      if (!name) {
        if (isAgent) {
          name = "Agent";
        } else if (isSelf) {
          name = "You";
        } else {
          name = "Unknown";
        }
      }
      allMessages.push({
        name,
        message: msg.message,
        timestamp: msg?.timestamp,
        isSelf: isSelf,
      });
    }
    allMessages.sort((a, b) => a.timestamp - b.timestamp);
    setMessages(allMessages);
  }, [transcripts, chatMessages, localParticipant, agentParticipant]);

  // reset transcripts and messages when name or activeTab changes
  useEffect(() => {
    setTranscripts([]);
    setMessages([]);
  }, [name, activeTab]);

  useDataChannel(activeTab !== 2 ? onDataReceived : undefined)

  const settingsTileContent = useMemo(() => {
    return (
      <div className="flex flex-col gap-4 h-full w-full items-start overflow-y-auto">
        <ConfigurationPanelItem title="Status">
          <div className="flex flex-col gap-2">
            <NameValueRow
              name="Room connected"
              value={
                roomState === ConnectionState.Connecting ? (
                  <LoadingSVG diameter={16} strokeWidth={2} />
                ) : (
                  roomState
                )
              }
              valueColor={
                roomState === ConnectionState.Connected
                  ? "cyan-500"
                  : "gray-500"
              }
            />
            <NameValueRow
              name="Agent connected"
              value={
                isAgentConnected ? (
                  "true"
                ) : roomState === ConnectionState.Connected ? (
                  <LoadingSVG diameter={12} strokeWidth={2} />
                ) : (
                  "false"
                )
              }
              valueColor={
                isAgentConnected
                  ? "cyan-500"
                  : "gray-500"
              }
            />
            <NameValueRow
              name="Agent status"
              value={
                agentState !== "offline" && agentState !== "speaking" ? (
                  <div className="flex gap-2 items-center">
                    <LoadingSVG diameter={12} strokeWidth={2} />
                    {agentState === "listening" && shouldShowChat ? "receiving" : agentState}
                  </div>
                ) : (
                  agentState
                )
              }
              valueColor={
                agentState === "speaking"
                  ? "cyan-500"
                  : "gray-500"
              }
            />
          </div>
        </ConfigurationPanelItem>

        {activeTab === 0 && localVideoTrack && (
          <ConfigurationPanelItem
            title="Camera"
            deviceSelectorKind="videoinput"
          >
            <div className="relative">
              <VideoTrack
                className="rounded-sm border border-gray-800 opacity-70 w-full"
                trackRef={localVideoTrack}
              />
            </div>
          </ConfigurationPanelItem>
        )}

        {needMicrophone && localMicTrack && (
          <ConfigurationPanelItem
            title="Microphone"
            deviceSelectorKind="audioinput"
          >
            <AudioInputTile frequencies={localMultibandVolume} />
          </ConfigurationPanelItem>
        )}
      </div>
    );
  }, [
    config.description,
    config.settings,
    config.show_qr,
    localParticipant,
    name,
    roomState,
    isAgentConnected,
    agentState,
    localVideoTrack,
    localMicTrack,
    localMultibandVolume,
    themeColors,
    setUserSettings,
  ]);

  const chatTileContent = useMemo(() => {
    return (
      <ChatTile
        messages={messages}
        accentColor="cyan"
        onSend={sendChat}
      />
    )
  }, [messages, sendChat])

  return (
    <div className="playground-container flex flex-col w-full h-full">
      <PlaygroundHeader
        title={config.title}
        logo={logo}
        githubLink={config.github_link}
        height={headerHeight}
        accentColor="cyan"
        tab={activeTab}
        connectionState={roomState}
        onConnectClicked={() =>
          onConnect(roomState === ConnectionState.Disconnected)
        }
      />

      <div
        className={"flex flex-col border rounded-sm border-gray-800 text-gray-500 bg-transparent"}
        style={{marginBottom: "6px"}}
      >
        <div
          className="flex items-center justify-start text-xs uppercase border-b border-b-gray-800 tracking-wider"
          style={{
            height: "32px",
          }}
        >
          {tabs.map((tab, index) => (
            <button
              key={index}
              className={`px-4 py-2 rounded-sm hover:bg-gray-800 hover:text-gray-300 border-r border-r-gray-800 ${
                index === activeTab
                  ? `bg-gray-900 text-gray-300`
                  : `bg-transparent text-gray-500`
              }`}
              disabled={!isTabActive}
              onClick={() => setActiveTab(index)}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="w-full grow flex flex-col justify-center items-center overflow-hidden">
        <PlaygroundTile
          padding={false}
          backgroundColor="gray-950"
          className={`w-full items-start ${shouldShowChat ? "h-auto mb-2" : "h-full overflow-y-auto"}`}
          childrenClassName="h-full grow items-start"
        >
          {settingsTileContent}
        </PlaygroundTile>

        {shouldShowChat && (
          <PlaygroundTile
          title="Chat"
          className="w-full h-full grow overflow-y-auto"
          childrenClassName="w-full h-full"
          >
            {chatTileContent}
          </PlaygroundTile>
        )}
      </div>
    </div>
  );
}
